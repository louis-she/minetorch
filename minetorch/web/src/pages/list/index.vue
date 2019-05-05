<template>
  <div class="list">
    <mpanel title="项目管理">
      <!-- 搜索 -->
      <div class="searchpanel">
        <div class="searchpanel-content">
          <el-form
            :inline="true"
            class="demo-form-inline">
            <el-form-item label="标题">
              <el-input v-model="searchForm.title" />
            </el-form-item>
            <el-form-item label="创建时间">
              <el-date-picker
                v-model="searchForm.time"
                type="date"
                placeholder="选择日期" />
            </el-form-item>
            <el-form-item label="状态">
              <el-select
                v-model="searchForm.status"
                placeholder="请选择">
                <el-option
                  v-for="(item, key) in statusText"
                  :key="`status-${key}`"
                  :label="item.text"
                  :value="key"/>
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button
                type="primary"
                @click="onSearch">查询</el-button>
            </el-form-item>
          </el-form>
        </div>
      </div>
      <!-- 新建 -->
      <el-button
        :style="{marginBottom: '20px'}"
        type="primary"
        @click="onAdd">新建</el-button>
      <!-- 列表 -->
      <el-table
        v-loading="loading"
        :data="tableData"
        stripe
        border
        style="width: 100%">
        <el-table-column
          prop="id"
          label="ID"
          align="center"
          width="100"/>
        <el-table-column
          prop="title"
          label="标题"
          align="center"/>
        <el-table-column
          label="状态"
          align="center">
          <template slot-scope="scope">
            {{ statusText[0].text }}
          </template>
        </el-table-column>
        <el-table-column
          label="创建时间"
          align="center">
          <template slot-scope="scope">
            {{ '2019-04-01 18:00' }}
          </template>
        </el-table-column>
        <el-table-column
          label="操作"
          align="center"
          width="300">
          <template slot-scope="scope">
            <el-button
              type="success"
              size="medium">开始训练</el-button>
            <el-button
              type="primary"
              size="medium">编辑</el-button>
            <el-button
              type="danger"
              size="medium">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
      <pagination
        :current-page="currentPage"
        :page-size="pageSize"
        :total="total"
        @change="changePage"/>
    </mpanel>
    <!-- 新建 -->
    <el-dialog
      :visible.sync="addVisible"
      :close-on-click-modal="false"
      class="add-form"
      title="新建"
      width="500px"
      @closed="addCancel">
      <el-form :model="addForm">
        <el-form-item label="test">
          <el-input
            v-model="addForm.test"
            autocomplete="off"/>
        </el-form-item>
      </el-form>
      <div
        slot="footer"
        class="dialog-footer">
        <el-button @click="addCancel">取 消</el-button>
        <el-button
          :loading="addForm.loading"
          type="primary"
          @click="addSure">确 定</el-button>
      </div>
    </el-dialog>
  </div>
</template>
<script>
import mpanel from 'components/_common/mpanel'
import pagination from 'components/_common/pagination'
import dateFormat from 'dateformat'

export default {
  components: {
    mpanel,
    pagination
  },
  data () {
    return {
      addVisible: false,
      loading: false,
      categoryList: [],
      addForm: {
        test: '',
        loading: false
      },
      searchForm: {
        title: '',
        time: '',
        status: ''
      },
      statusText: {
        0: {
          text: '未运行'
        },
        1: {
          text: '运行中'
        }
      },
      tableData: [],
      currentPage: 1,
      pageSize: 10,
      total: 1
    }
  },
  mounted () {
    // this.getData()
  },
  methods: {
    dateFormat,
    // 获取列表
    async getData () {
      this.loading = true
      const category = await this.ajax.post('/api/xxxx', {})
      console.log(category)
    },
    // 搜索
    onSearch () {
      this.currentPage = 1
      this.getData()
    },
    // 新建
    onAdd () {
      this.addVisible = true
    },
    addCancel () {
      this.addVisible = false
      this.addForm = {
        titleSc: '',
        titleTc: '',
        titleEn: '',
        category: '',
        loading: false
      }
    },
    addSure () {
      this.addForm.loading = true
    },
    // 翻页
    changePage (page, size) {
      this.currentPage = page
      this.pageSize = size
      this.getData()
    }
  }
}
</script>
<style lang="scss">
.searchpanel {
  padding: 20px;
  background-color: #fafafc;
  border: 1px solid #e5e5e5;
  margin-bottom: 20px;
  .el-form-item {
    margin-bottom: 0;
  }
}
.add-form {
  .el-select {
    width: 100%;
  }
}
</style>
