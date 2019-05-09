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

    <mpanel :title="componentName">
      <!-- 列表 -->
      <el-table v-loading="loading" :data="components" stripe style="width: 100%">
        <el-table-column :label="tableTitle" prop="name" align="center"/>
        <el-table-column prop="description" label="description" align="center" />
        <el-table-column label="operations" align="center" width="300">
          <template slot-scope="scope">
            <el-button size="mini" @click="handleSelectClicked(scope)">
              select
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </mpanel>
    <el-dialog :visible.sync="openDialog" :before-close="handleClose" title="Tips">
      <dynamic-form />
      <span slot="footer" class="dialog-footer">
        <el-button @click="dialogVisible = false">Cancel</el-button>
        <el-button type="primary" @click="dialogVisible = false">Confirm</el-button>
      </span>
    </el-dialog>
  </div>
</template>
<script>
import dateFormat from 'dateformat'
import moment from 'moment'
import Vue from 'vue'
import dynamicForm from 'components/dynamic-form'

export default {
  components: {
    'dynamic-form': dynamicForm
  },
  data () {
    return {
      componentName: '',
      components: [],
      openDialog: false
    }
  },
  computed: {
    tableTitle() {
      return `${this.componentName} Name`
    }
  },
  mounted () {
    this.getComponents()
  },
  methods: {
    async getComponents() {
      this.loading = true
      this.components = await this.ajax.get(`/api/experiments/${this.$route.params.experimentId}/datasets`)
      this.loading = false
    },

    handleSelectClicked(scope) {
      this.openDialog = true
    }
  }
}
</script>
<style lang="scss">
.hide {
  display: none;
}
</style>
