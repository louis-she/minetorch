<template>
  <div class="experiment-dataset">
    <pre class="hide">
      <code>
        1. 请求 /api/$components (这里是datasets) 接口获取所有可选择的 $components
        2. 用户选择某个 component 后，弹窗显示配置表单
        3. 表单被正确提交后，进到下一个 Component 的步骤
        4. 如果用户回退到当前 Component 的配置，显示用户已经配置好信息，然后底部两个按钮「修改」「选择其他的$resource」
           点击「修改」后，弹窗显示配置表单，保存后刷新页面信息（不要进到下一步）。点击 「选择其他$resource」后，
           重复步骤 1
      </code>
    </pre>
    <!-- 列表 -->
    <el-table v-loading="loading" :data="components" :row-class-name="tableRowClassName"
              stripe style="width: 100%">
      <el-table-column :label="tableTitle" prop="name" align="center"/>
      <el-table-column prop="description" label="description" align="center" />
      <el-table-column label="operations" align="center" width="300">
        <template slot-scope="scope">
          <el-button v-if="scope.$index !== selectedRowIndex" size="mini" @click="handleSelectClicked(scope)">
            select
          </el-button>
          <el-button v-else-if="selectedComponentConfigurable" size="mini" type="success"
                     @click="handleSelectClicked(scope, 'edit')">
            edit
          </el-button>
          <el-tooltip v-else content="Top center" placement="top">
            <el-button disabled size="mini" type="success">edit</el-button>
          </el-tooltip>
        </template>
      </el-table-column>
    </el-table>
    <el-dialog :visible.sync="openDialog" :title="dialogTitle">
      <dynamic-form :extra-data="{name: selectComponentName}" :schema="schema" :submit-url="url"
                    @dynamic-form:cancel="openDialog=false"
                    @dynamic-form:success="handleFormSubmitted" />
    </el-dialog>
    <el-dialog :visible.sync="openEditDialog" :title="editDialogTitle">
      <dynamic-form :schema="schema" :submit-url="selectedUrl" submit-button="Update" method="patch"
                    @dynamic-form:cancel="openEditDialog=false"
                    @dynamic-form:success="handleEditFormSubmitted" />
    </el-dialog>
  </div>
</template>
<script>
import dynamicForm from 'components/dynamic-form'
import { camelCase } from 'lodash'

export default {
  components: {
    'dynamic-form': dynamicForm
  },
  data () {
    return {
      componentName: '',
      components: [],
      selectedComponent: {},
      openDialog: false,
      openEditDialog: false,
      dialogTitle: '',
      schema: [],
      experimentId: '',
      selectComponentName: ''
    }
  },
  computed: {
    tableTitle() {
      return `${this.componentName} name`
    },
    url() {
      return `/api/experiments/${this.experimentId}/${this.pluralComponentName}`
    },
    selectedUrl() {
      return `/api/experiments/${this.experimentId}/${this.pluralComponentName}/selected`
    },
    selectedRowIndex() {
      return this.components.findIndex(component => component.name === this.selectedComponent.name)
    },
    editDialogTitle() {
      return `Edit ${this.dialogTitle}`
    },
    pluralComponentName() {
      if (this.componentName.slice(-1) === 's') {
        return `${this.componentName}es`
      }
      return `${this.componentName}s`
    },
    selectedComponentConfigurable() {
      return !!Object.keys(this.selectedComponent.settings).length
    }
  },
  mounted () {
    this.componentName = this.$route.params.componentName
    this.experimentId = this.$route.params.experimentId
    this.getComponents()
    this.getSelectedComponent()
  },
  methods: {
    async getComponents() {
      this.loading = true
      this.components = await this.ajax.get(this.url)
      this.loading = false
    },

    tableRowClassName({row, rowIndex}) {
      if (rowIndex === this.selectedRowIndex) {
        return 'success-row'
      }
    },

    handleFormSubmitted(response) {
      this.openDialog = false
      this.selectedComponent = response
      this.navigateToNextStep()
    },

    navigateToNextStep() {
      this.$router.push({
        name: 'EditExperimentComponent',
        params: {
          experimentId: this.experimentId,
          componentName: this.steps[this.steps.indexOf(this.componentName) + 1]
        }
      })
    },

    handleEditFormSubmitted(response) {
      this.openEditDialog = false
      this.selectedComponent = response
      this.navigateToNextStep()
    },

    async getSelectedComponent() {
      this.selectedComponent = await this.ajax.get(this.selectedUrl)
    },

    async handleSelectClicked(scope, type = 'create') {
      const component = this.components[scope.$index]
      const options = component.options.slice()
      if (!options.length) {
        // the selected component has no configurable options
        // create it directly
        this.selectedComponent = await this.ajax.post(this.url, {name: component.name})
        this.navigateToNextStep()
        return
      }

      if (this.selectedComponent.name === component.name) {
        // this is **the** selected component, should override default value
        options.forEach((option) => {
          const value = this.selectedComponent.settings[camelCase(option.name)]
          option.settings.default = value === 'true' ? true : value
        })
      }

      const schema = options.map((option) => {
        return {
          label: option.settings.label || option.name,
          type: option.settings.type || 'string',
          name: option.name,
          ...option.settings
        }
      })

      this.dialogTitle = component.name
      this.schema = schema
      this.selectComponentName = component.name
      if (type === 'create') {
        this.openDialog = true
      } else {
        this.openEditDialog = true
      }
    }
  }
}
</script>
<style lang="scss">
.hide {
  display: none;
}

.el-table .success-row {
  background: #f0f9eb;
}
</style>
